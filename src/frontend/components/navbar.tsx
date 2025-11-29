"use client";

import { usePathname } from "next/navigation";
import {
  Navbar as HeroUINavbar,
  NavbarContent,
  NavbarMenu,
  NavbarMenuToggle,
  NavbarBrand,
  NavbarItem,
  NavbarMenuItem,
} from "@heroui/navbar";
import { Link } from "@heroui/link";
import { link as linkStyles } from "@heroui/theme";
import NextLink from "next/link";
import clsx from "clsx";

import { siteConfig } from "@/config/site";
import { ThemeSwitch } from "@/components/theme-switch";
import { Logo } from "@/components/icons";
import { CustomSearchInput } from "@/components/search-input";

export const Navbar = () => {
  const pathname = usePathname();
  const isHome = pathname === "/";

  return (
    <HeroUINavbar maxWidth="xl" position="sticky">
      
      {/* Logo */}
      <NavbarContent className="basis-auto" justify="start">
        <NavbarBrand as="li" className="gap-3 max-w-fit pr-6">
          <NextLink className="group flex justify-start items-center gap-1" href="/">
            <div className="transition-all duration-300 group-hover:drop-shadow-[0_0_12px_rgba(250,204,21,0.9)]">
              <Logo />
            </div>
            <p className="font-bold text-inherit transition-colors duration-300 group-hover:text-yellow-500">
              Lemon Nipis
            </p>
          </NextLink>
        </NavbarBrand>
      </NavbarContent>

      {/* Search Bar */}
      <NavbarContent className="hidden sm:flex basis-1/2 max-w-[500px] flex-grow" justify="center">
        <NavbarItem className="w-full">
           {!isHome && <CustomSearchInput />}
        </NavbarItem>
      </NavbarContent>

      {/* Menu & Dark / Light Mode */}
      <NavbarContent className="basis-auto" justify="end">
        <ul className="hidden md:flex gap-2 justify-start mr-4">
          {siteConfig.navItems.map((item) => {
            const isActive = pathname === item.href;

            return (
              <NavbarItem key={item.href}>
                <NextLink
                  className={clsx(
                    linkStyles({ color: "foreground" }),
                    "px-4 py-1.5 rounded-lg border text-sm font-medium transition-colors duration-200", 
                    
                    isActive 
                      ? "bg-yellow-400/20 border-yellow-400/50 text-yellow-600 dark:text-yellow-400 font-bold" 
                      : "border-transparent hover:bg-default-100"
                  )}
                  color="foreground"
                  href={item.href}
                >
                  {item.label}
                </NextLink>
              </NavbarItem>
            );
          })}
        </ul>
        <ThemeSwitch />
        <NavbarMenuToggle className="md:hidden" /> 
      </NavbarContent>

      {/* Menu kalo dikecilin layarnya */}
      <NavbarMenu>
        <div className="mt-2 mb-4">
          <CustomSearchInput />
        </div>
        
        <div className="mx-4 mt-2 flex flex-col gap-2 items-end">
          {siteConfig.navMenuItems.map((item, index) => {
             const isActive = pathname === item.href;
             
             return (
                <NavbarMenuItem key={`${item}-${index}`}>
                  <Link
                    as={NextLink}
                    // Style Mobile Kuning
                    className={clsx(
                        "w-full px-4 py-2 rounded-md block text-right transition-colors",
                        isActive 
                        ? "bg-yellow-400/10 text-yellow-600 font-bold" 
                        : "text-foreground"
                    )}
                    href={item.href}
                    size="lg"
                  >
                    {item.label}
                  </Link>
                </NavbarMenuItem>
             );
          })}
        </div>
      </NavbarMenu>

    </HeroUINavbar>
  );
};